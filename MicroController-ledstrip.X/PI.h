#ifndef PI_H
#define	PI_H

#include <xc.h> // include processor files - each processor file is guarded.

#ifdef	__cplusplus
extern "C" {
#endif /* __cplusplus */

    void PI(void);
    
    uint8_t PI_GetSensorHeight(void);
    
    void PI_SetSetpoint(uint8_t value);
    
    void PI_SetKp(float value);
    
    void PI_SetKi(float value);

#ifdef	__cplusplus
}
#endif /* __cplusplus */

#endif	/* PI_H */

